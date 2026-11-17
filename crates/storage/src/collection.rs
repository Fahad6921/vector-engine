//! A collection of vectors with external u64 ids and opaque payloads.
//!
//! Writes go to the WAL first, then into an in-memory HNSW (the memtable). When the
//! memtable reaches `flush_threshold` points it is written out as an immutable
//! segment. Searches query every segment plus the memtable and merge the results.
//! Deletes and overwrites set a bit in the owning segment's deletion bitmap;
//! deleted points stay in the graph (they still help navigation) until `compact`.
//!
//! Directory layout:
//!   manifest.json   config, live segment ids; replacing it commits a flush/compaction
//!   wal.log         writes since the last flush
//!   seg-N.*         segment files, see `segment.rs`
//!
//! Crash safety: segment files are written before the manifest that references
//! them, and the WAL is only emptied after the manifest is replaced. Recovery loads
//! the segments the manifest lists, deletes any other segment files, and replays
//! the WAL. Replaying a write that already reached a segment is harmless: it
//! overwrites the point with the same value.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use vecengine::text::{rrf, tokenize, CorpusStats};
use vecengine::{HnswParams, Metric};

use crate::filter::{validate_payload, Filter, PayloadIndex};
use crate::segment::{build_text_index, parse_segment_file, text_of, write_atomic, Segment};
use crate::wal::{Record, Wal};

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub metric: Metric,
    pub m: usize,
    pub ef_construction: usize,
    /// Memtable size at which it is written out as a segment.
    pub flush_threshold: usize,
    /// fsync the WAL on every write. Off trades durability on power loss for speed.
    pub sync: bool,
    /// Payload field to index for BM25 keyword and hybrid search.
    pub text_field: Option<String>,
}

impl Config {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            m: 16,
            ef_construction: 200,
            flush_threshold: 100_000,
            sync: true,
            text_field: None,
        }
    }

    fn params(&self) -> HnswParams {
        HnswParams {
            m: self.m,
            ef_construction: self.ef_construction,
            ..Default::default()
        }
    }
}

/// How a filtered search runs within each segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterStrategy {
    /// Exact scan of the matching points when few match, graph walk otherwise.
    #[default]
    Auto,
    /// Always walk the graph, skipping non-matching points.
    Graph,
    /// Always score every matching point exactly.
    BruteForce,
}

/// Candidates taken from each of vector and keyword search before fusion.
const HYBRID_DEPTH: usize = 100;
/// The usual RRF constant (Cormack et al., 2009).
const RRF_C: f32 = 60.0;

/// Matching-point count at or below which `Auto` scans exactly.
fn brute_force_limit(segment_len: usize) -> usize {
    segment_len / 10
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    pub live: usize,
    pub segments: usize,
    pub memtable: usize,
    pub deleted: usize,
}

pub struct Collection {
    dir: PathBuf,
    config: Config,
    /// On-disk segments, oldest first; the last entry is always the memtable.
    parts: Vec<Segment>,
    wal: Wal,
    /// External id -> (part index, internal id) for every live point.
    locations: HashMap<u64, (u32, u32)>,
    next_segment: u64,
}

impl Collection {
    /// Opens the collection in `dir`, creating it if needed. For an existing
    /// collection, `dim`, `metric`, `m` and `ef_construction` must match.
    pub fn open(dir: impl AsRef<Path>, config: Config) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if config.dim == 0 || config.m < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dim must be > 0 and m >= 2",
            ));
        }
        fs::create_dir_all(&dir)?;

        let manifest = dir.join("manifest.json");
        let (segment_ids, next_segment) = if manifest.exists() {
            read_manifest(&manifest, &config)?
        } else {
            write_manifest(&manifest, &config, &[], 1)?;
            (Vec::new(), 1)
        };

        remove_orphans(&dir, &segment_ids)?;
        let mut parts = segment_ids
            .iter()
            .map(|&id| Segment::load(&dir, id, config.text_field.as_deref()))
            .collect::<io::Result<Vec<_>>>()?;
        parts.push(Segment::empty(config.dim, config.metric, config.params()));

        let mut locations = HashMap::new();
        for (p, part) in parts.iter().enumerate() {
            for (i, &id) in part.ids.iter().enumerate() {
                if part.is_live(i as u32) {
                    locations.insert(id, (p as u32, i as u32));
                }
            }
        }

        let (wal, records) = Wal::open(&dir.join("wal.log"), config.sync)?;
        let mut c = Self {
            dir,
            config,
            parts,
            wal,
            locations,
            next_segment,
        };
        c.apply(records);
        Ok(c)
    }

    /// The config an existing collection was created with, read from its manifest.
    /// Runtime-only settings (`flush_threshold`, `sync`) come back as defaults.
    pub fn stored_config(dir: impl AsRef<Path>) -> io::Result<Config> {
        Ok(parse_manifest(&dir.as_ref().join("manifest.json"))?.config)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    pub fn stats(&self) -> Stats {
        let (mem, segs) = self.parts.split_last().unwrap();
        Stats {
            live: self.len(),
            segments: segs.len(),
            memtable: mem.ids.len(),
            deleted: self.parts.iter().map(|p| p.deleted.count()).sum(),
        }
    }

    pub fn upsert(&mut self, id: u64, vector: &[f32], payload: &[u8]) -> io::Result<()> {
        self.write(vec![Record::Upsert {
            id,
            vector: vector.to_vec(),
            payload: payload.to_vec(),
        }])
    }

    pub fn delete(&mut self, id: u64) -> io::Result<()> {
        self.write(vec![Record::Delete { id }])
    }

    /// Applies a batch of writes with a single WAL append (one fsync).
    pub fn write(&mut self, records: Vec<Record>) -> io::Result<()> {
        for r in &records {
            if let Record::Upsert { vector, payload, .. } = r {
                validate_payload(payload).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
                if vector.len() != self.config.dim {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("expected dim {}, got {}", self.config.dim, vector.len()),
                    ));
                }
                if !vector.iter().all(|x| x.is_finite()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "vector has NaN or inf",
                    ));
                }
            }
        }
        self.wal.append(&records)?;
        self.apply(records);
        if self.parts.last().unwrap().ids.len() >= self.config.flush_threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// Applies records to the memtable without logging them.
    fn apply(&mut self, records: Vec<Record>) {
        let mem_index = self.parts.len() - 1;
        let mut vectors = Vec::new();
        for r in records {
            match r {
                Record::Upsert { id, vector, payload } => {
                    self.mark_deleted(id);
                    let mem = &mut self.parts[mem_index];
                    let internal = mem.ids.len() as u32;
                    mem.ids.push(id);
                    mem.payload_index.add(internal, &payload);
                    if let Some(t) = self
                        .config
                        .text_field
                        .as_deref()
                        .and_then(|f| text_of(&payload, f))
                    {
                        mem.text.add(internal, &t);
                    }
                    mem.payloads.push(payload);
                    vectors.extend_from_slice(&vector);
                    self.locations.insert(id, (mem_index as u32, internal));
                }
                Record::Delete { id } => self.mark_deleted(id),
            }
        }
        let mem = &mut self.parts[mem_index];
        mem.index.add_batch(&vectors);
        debug_assert_eq!(mem.index.len(), mem.ids.len());
    }

    fn mark_deleted(&mut self, id: u64) {
        if let Some((p, i)) = self.locations.remove(&id) {
            let part = &mut self.parts[p as usize];
            part.deleted.set(i as usize);
            part.deleted_dirty = true;
        }
    }

    /// The stored vector and payload for `id`. Cosine vectors come back normalised.
    pub fn get(&self, id: u64) -> Option<(&[f32], &[u8])> {
        let &(p, i) = self.locations.get(&id)?;
        let part = &self.parts[p as usize];
        Some((part.index.vector(i), &part.payloads[i as usize]))
    }

    /// The `k` nearest live points across all segments, closest first.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Hit> {
        self.search_filtered(query, k, ef, None, FilterStrategy::Auto)
    }

    /// Like `search`, restricted to points whose payload matches `filter`.
    ///
    /// Per segment, when few points match, the graph walk would spend most of its
    /// time on rejected nodes, so `Auto` scores the matching points exactly
    /// instead. The cut-off was picked from `examples/filter_bench.rs`.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
        strategy: FilterStrategy,
    ) -> Vec<Hit> {
        let mut hits: Vec<Hit> = Vec::new();
        for part in &self.parts {
            let n = part.ids.len();
            if part.live_count() == 0 {
                continue;
            }
            let res = match filter {
                None => part.index.search_filtered(query, k, ef, |i| part.is_live(i)),
                Some(f) => {
                    let allowed = part.payload_index.eval(f, n).and_not(&part.deleted);
                    let matching = allowed.count();
                    if matching == 0 {
                        continue;
                    }
                    let brute = match strategy {
                        FilterStrategy::Auto => matching <= brute_force_limit(n),
                        FilterStrategy::Graph => false,
                        FilterStrategy::BruteForce => true,
                    };
                    if brute {
                        part.index.search_subset(query, k, allowed.iter())
                    } else {
                        part.index
                            .search_filtered(query, k, ef, |i| allowed.get(i as usize))
                    }
                }
            };
            hits.extend(res.iter().map(|n| Hit {
                id: part.ids[n.id as usize],
                distance: n.distance,
            }));
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }

    /// BM25 keyword search over the configured text field. Hits carry the negated
    /// BM25 score as `distance`, so lower is still better. Empty if the collection
    /// has no text field.
    pub fn search_text(&self, query: &str, k: usize, filter: Option<&Filter>) -> Vec<Hit> {
        let terms = tokenize(query);
        let mut stats = CorpusStats::default();
        for part in &self.parts {
            stats.merge(&part.text.stats(&terms));
        }
        let mut hits = Vec::new();
        for part in &self.parts {
            if part.live_count() == 0 {
                continue;
            }
            let allowed = filter.map(|f| part.payload_index.eval(f, part.ids.len()));
            let res = part.text.search(&terms, k, &stats, |d| {
                part.is_live(d) && allowed.as_ref().is_none_or(|a| a.get(d as usize))
            });
            hits.extend(res.iter().map(|n| Hit {
                id: part.ids[n.id as usize],
                distance: n.distance,
            }));
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }

    /// Vector and keyword search fused with reciprocal rank fusion over the top
    /// `HYBRID_DEPTH` of each. `distance` is the negated fused score.
    pub fn search_hybrid(
        &self,
        vector: &[f32],
        text: &str,
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
    ) -> Vec<Hit> {
        let depth = HYBRID_DEPTH.max(k);
        let by_vector: Vec<u64> = self
            .search_filtered(vector, depth, ef.max(depth), filter, FilterStrategy::Auto)
            .iter()
            .map(|h| h.id)
            .collect();
        let by_text: Vec<u64> = self
            .search_text(text, depth, filter)
            .iter()
            .map(|h| h.id)
            .collect();
        rrf(&[by_vector, by_text], RRF_C)
            .into_iter()
            .take(k)
            .map(|(id, score)| Hit { id, distance: -score })
            .collect()
    }

    /// Writes the memtable out as a segment, persists deletion bitmaps and empties
    /// the WAL.
    pub fn flush(&mut self) -> io::Result<()> {
        let mem_index = self.parts.len() - 1;
        let mem_has_data = self.parts[mem_index].live_count() > 0;
        if mem_has_data {
            let id = self.next_segment;
            self.next_segment += 1;
            self.parts[mem_index].write_new(&self.dir, id)?;
        }
        for part in &mut self.parts {
            part.write_deletes(&self.dir)?;
        }

        // Points in a memtable that was entirely deleted are simply dropped.
        if !mem_has_data {
            self.parts.pop();
        }
        self.parts.push(Segment::empty(
            self.config.dim,
            self.config.metric,
            self.config.params(),
        ));

        let ids = self.segment_ids();
        write_manifest(
            &self.dir.join("manifest.json"),
            &self.config,
            &ids,
            self.next_segment,
        )?;
        self.wal.reset()
    }

    /// Rebuilds all segments into one, dropping deleted points.
    pub fn compact(&mut self) -> io::Result<()> {
        self.flush()?;
        let old = self.segment_ids();
        if old.is_empty() {
            return Ok(());
        }

        let mut merged = Segment::empty(self.config.dim, self.config.metric, self.config.params());
        let mut vectors = Vec::with_capacity(self.len() * self.config.dim);
        for part in &self.parts {
            for (i, &id) in part.ids.iter().enumerate() {
                if part.is_live(i as u32) {
                    merged.ids.push(id);
                    merged.payloads.push(part.payloads[i].clone());
                    vectors.extend_from_slice(part.index.vector(i as u32));
                }
            }
        }
        merged.index.add_batch(&vectors);
        merged.payload_index = PayloadIndex::build(merged.payloads.iter().map(Vec::as_slice));
        merged.text = build_text_index(&merged.payloads, self.config.text_field.as_deref());

        let mut parts = Vec::new();
        if !merged.ids.is_empty() {
            let id = self.next_segment;
            self.next_segment += 1;
            merged.write_new(&self.dir, id)?;
            parts.push(merged);
        }
        parts.push(Segment::empty(
            self.config.dim,
            self.config.metric,
            self.config.params(),
        ));
        self.parts = parts;

        let ids = self.segment_ids();
        write_manifest(
            &self.dir.join("manifest.json"),
            &self.config,
            &ids,
            self.next_segment,
        )?;
        for id in old {
            Segment::remove_files(&self.dir, id);
        }
        self.locations = self.parts[0]
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, (0, i as u32)))
            .collect();
        if self.parts.len() == 1 {
            self.locations.clear();
        }
        Ok(())
    }

    fn segment_ids(&self) -> Vec<u64> {
        self.parts.iter().filter_map(|p| p.id).collect()
    }
}

fn write_manifest(path: &Path, config: &Config, segments: &[u64], next_segment: u64) -> io::Result<()> {
    let json = serde_json::json!({
        "version": 1,
        "dim": config.dim,
        "metric": config.metric.name(),
        "m": config.m,
        "ef_construction": config.ef_construction,
        "text_field": config.text_field,
        "segments": segments,
        "next_segment": next_segment,
    });
    write_atomic(path, serde_json::to_string_pretty(&json).unwrap().as_bytes())
}

struct Manifest {
    config: Config,
    segments: Vec<u64>,
    next_segment: u64,
}

fn parse_manifest(path: &Path) -> io::Result<Manifest> {
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("manifest: {msg}"));
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(path)?).map_err(|e| bad(&e.to_string()))?;
    let field = |k: &str| json.get(k).ok_or_else(|| bad(&format!("missing {k}")));
    let num = |k: &str| {
        field(k)?
            .as_u64()
            .ok_or_else(|| bad(&format!("{k} is not a number")))
    };
    let metric = field("metric")?
        .as_str()
        .and_then(Metric::from_name)
        .ok_or_else(|| bad("unknown metric"))?;
    let config = Config {
        m: num("m")? as usize,
        ef_construction: num("ef_construction")? as usize,
        text_field: json.get("text_field").and_then(|v| v.as_str()).map(String::from),
        ..Config::new(num("dim")? as usize, metric)
    };
    let segments = field("segments")?
        .as_array()
        .ok_or_else(|| bad("segments is not a list"))?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| bad("bad segment id")))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(Manifest {
        config,
        segments,
        next_segment: num("next_segment")?,
    })
}

fn read_manifest(path: &Path, config: &Config) -> io::Result<(Vec<u64>, u64)> {
    let m = parse_manifest(path)?;
    let key = |c: &Config| (c.dim, c.metric, c.m, c.ef_construction, c.text_field.clone());
    if key(&m.config) != key(config) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "collection was created with (dim, metric, m, ef_construction, text_field) = {:?}, opened with {:?}",
                key(&m.config),
                key(config)
            ),
        ));
    }
    Ok((m.segments, m.next_segment))
}

/// Deletes segment files the manifest doesn't reference (left by a crash during
/// flush or compaction) and stray temp files.
fn remove_orphans(dir: &Path, live: &[u64]) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let orphan = name.ends_with(".tmp") || parse_segment_file(name).is_some_and(|id| !live.contains(&id));
        if orphan {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vecengine-coll-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// A distinct pseudo-random vector per id.
    fn vec_for(id: u64, dim: usize) -> Vec<f32> {
        (0..dim as u64)
            .map(|j| {
                let mut z = id.wrapping_mul(0x9e3779b97f4a7c15) ^ j.wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 31)).wrapping_mul(0x94d049bb133111eb);
                (z >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    fn config(flush: usize) -> Config {
        Config {
            flush_threshold: flush,
            ..Config::new(8, Metric::L2)
        }
    }

    #[test]
    fn upsert_search_get_delete() {
        let dir = temp_dir("basic");
        let mut c = Collection::open(&dir, config(1000)).unwrap();
        for id in 0..100 {
            c.upsert(id, &vec_for(id, 8), format!(r#"{{"n":{id}}}"#).as_bytes())
                .unwrap();
        }
        assert_eq!(c.len(), 100);
        assert_eq!(c.search(&vec_for(42, 8), 1, 50)[0].id, 42);
        assert_eq!(c.get(42).unwrap().1, br#"{"n":42}"#);

        c.delete(42).unwrap();
        assert!(c.get(42).is_none());
        assert_ne!(c.search(&vec_for(42, 8), 1, 50)[0].id, 42);

        c.upsert(7, &vec_for(500, 8), br#"{"v":"new"}"#).unwrap();
        assert_eq!(c.len(), 99);
        assert_eq!(c.get(7).unwrap().1, br#"{"v":"new"}"#);
        assert_eq!(c.search(&vec_for(500, 8), 1, 50)[0].id, 7);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn survives_reopen_with_and_without_flush() {
        let dir = temp_dir("reopen");
        {
            let mut c = Collection::open(&dir, config(30)).unwrap();
            for id in 0..100 {
                c.upsert(id, &vec_for(id, 8), format!(r#"{{"n":{id}}}"#).as_bytes())
                    .unwrap();
            }
            c.delete(3).unwrap(); // in a flushed segment
            c.delete(95).unwrap(); // in the memtable
            c.upsert(10, &vec_for(1000, 8), br#"{"v":"moved"}"#).unwrap();
            assert!(c.stats().segments >= 3);
        }
        let c = Collection::open(&dir, config(30)).unwrap();
        assert_eq!(c.len(), 98);
        assert!(c.get(3).is_none() && c.get(95).is_none());
        assert_eq!(c.get(10).unwrap().1, br#"{"v":"moved"}"#);
        assert_eq!(c.get(50).unwrap().0, vec_for(50, 8).as_slice());
        assert_eq!(c.search(&vec_for(1000, 8), 1, 50)[0].id, 10);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compaction_drops_deleted_points() {
        let dir = temp_dir("compact");
        let mut c = Collection::open(&dir, config(25)).unwrap();
        for id in 0..100 {
            c.upsert(id, &vec_for(id, 8), b"").unwrap();
        }
        for id in (0..100).step_by(2) {
            c.delete(id).unwrap();
        }
        c.compact().unwrap();
        let s = c.stats();
        assert_eq!((s.live, s.segments, s.deleted, s.memtable), (50, 1, 0, 0));
        let files = fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, 4, "manifest, wal, one segment's .hnsw and .meta");

        drop(c);
        let c = Collection::open(&dir, config(25)).unwrap();
        assert_eq!(c.len(), 50);
        assert_eq!(c.search(&vec_for(51, 8), 1, 50)[0].id, 51);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deleting_everything_then_compacting() {
        let dir = temp_dir("empty");
        let mut c = Collection::open(&dir, config(10)).unwrap();
        for id in 0..20 {
            c.upsert(id, &vec_for(id, 8), b"").unwrap();
        }
        for id in 0..20 {
            c.delete(id).unwrap();
        }
        c.compact().unwrap();
        assert!(c.is_empty());
        assert!(c.search(&vec_for(1, 8), 5, 50).is_empty());
        c.upsert(1, &vec_for(1, 8), b"").unwrap();
        assert_eq!(c.search(&vec_for(1, 8), 5, 50)[0].id, 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn filtered_search_across_segments() {
        let dir = temp_dir("filtered");
        let mut c = Collection::open(&dir, config(150)).unwrap();
        for id in 0..400u64 {
            let payload = format!(
                r#"{{"parity": "{}", "n": {id}}}"#,
                if id % 2 == 0 { "even" } else { "odd" }
            );
            c.upsert(id, &vec_for(id, 8), payload.as_bytes()).unwrap();
        }
        assert!(c.stats().segments >= 2 && c.stats().memtable > 0);
        c.delete(10).unwrap();

        let odd = Filter::parse(r#"{"field": "parity", "eq": "odd"}"#).unwrap();
        let rare = Filter::parse(r#"{"field": "n", "range": {"gte": 10, "lte": 12}}"#).unwrap();
        for strategy in [
            FilterStrategy::Auto,
            FilterStrategy::Graph,
            FilterStrategy::BruteForce,
        ] {
            let hits = c.search_filtered(&vec_for(5, 8), 10, 100, Some(&odd), strategy);
            assert_eq!(hits.len(), 10);
            assert!(hits.iter().all(|h| h.id % 2 == 1), "{strategy:?}");
            assert_eq!(hits[0].id, 5);

            // 10 was deleted, so only 11 and 12 match.
            let hits = c.search_filtered(&vec_for(10, 8), 10, 100, Some(&rare), strategy);
            let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
            ids.sort();
            assert_eq!(ids, vec![11, 12], "{strategy:?}");
        }

        // Filters still work after reopening (index rebuilt from segment payloads).
        drop(c);
        let c = Collection::open(&dir, config(150)).unwrap();
        let hits = c.search_filtered(&vec_for(10, 8), 10, 100, Some(&rare), FilterStrategy::Auto);
        assert_eq!(hits.len(), 2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn text_and_hybrid_search() {
        let dir = temp_dir("text");
        let cfg = || Config {
            text_field: Some("body".into()),
            ..config(3)
        };
        let docs = [
            "rust vector database",
            "python data science",
            "rust compiler internals",
            "baking sourdough bread",
            "vector search with hnsw graphs",
        ];
        let mut c = Collection::open(&dir, cfg()).unwrap();
        for (i, d) in docs.iter().enumerate() {
            let payload = format!(
                r#"{{"body": "{d}", "lang": "{}"}}"#,
                if i % 2 == 0 { "en" } else { "fr" }
            );
            c.upsert(i as u64, &vec_for(i as u64, 8), payload.as_bytes())
                .unwrap();
        }
        assert!(c.stats().segments >= 1 && c.stats().memtable > 0);

        let ids = |hits: Vec<Hit>| hits.iter().map(|h| h.id).collect::<Vec<_>>();
        assert_eq!(ids(c.search_text("rust", 10, None)), vec![0, 2]);
        assert_eq!(ids(c.search_text("vector", 10, None)).len(), 2);
        assert!(c.search_text("quantum", 10, None).is_empty());
        let fr = Filter::parse(r#"{"field": "lang", "eq": "fr"}"#).unwrap();
        assert_eq!(ids(c.search_text("rust python", 10, Some(&fr))), vec![1]);

        // Hybrid: doc 4 is the vector match, doc 0 the keyword match; both rank.
        let hybrid = ids(c.search_hybrid(&vec_for(4, 8), "rust database", 3, 50, None));
        assert!(hybrid.contains(&4) && hybrid.contains(&0), "{hybrid:?}");

        c.delete(0).unwrap();
        assert_eq!(ids(c.search_text("rust", 10, None)), vec![2]);
        drop(c);

        let c = Collection::open(&dir, cfg()).unwrap();
        assert_eq!(ids(c.search_text("rust", 10, None)), vec![2]);
        assert!(
            Collection::open(&dir, config(3)).is_err(),
            "text_field must match"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn stored_config_roundtrips() {
        let dir = temp_dir("stored");
        let cfg = Config {
            m: 8,
            text_field: Some("body".into()),
            ..Config::new(5, Metric::Cosine)
        };
        Collection::open(&dir, cfg.clone()).unwrap();
        let stored = Collection::stored_config(&dir).unwrap();
        assert_eq!(
            (
                stored.dim,
                stored.metric,
                stored.m,
                stored.ef_construction,
                stored.text_field.clone()
            ),
            (5, Metric::Cosine, 8, 200, Some("body".into()))
        );
        Collection::open(&dir, stored).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_non_object_payloads() {
        let dir = temp_dir("payloads");
        let mut c = Collection::open(&dir, config(10)).unwrap();
        assert!(c.upsert(1, &vec_for(1, 8), b"[1, 2]").is_err());
        assert!(c.upsert(1, &vec_for(1, 8), b"not json").is_err());
        assert!(c.upsert(1, &vec_for(1, 8), b"").is_ok());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_bad_input_and_mismatched_config() {
        let dir = temp_dir("bad");
        let mut c = Collection::open(&dir, config(10)).unwrap();
        assert!(c.upsert(1, &[1.0; 3], b"").is_err());
        assert!(c.upsert(1, &[f32::NAN; 8], b"").is_err());
        assert!(c.is_empty());
        drop(c);
        assert!(Collection::open(&dir, Config::new(16, Metric::L2)).is_err());
        assert!(Collection::open(&dir, Config::new(8, Metric::Cosine)).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn removes_orphaned_segment_files() {
        let dir = temp_dir("orphans");
        Collection::open(&dir, config(10)).unwrap();
        fs::write(dir.join("seg-000099.hnsw"), b"junk").unwrap();
        fs::write(dir.join("manifest.tmp"), b"junk").unwrap();
        Collection::open(&dir, config(10)).unwrap();
        assert!(!dir.join("seg-000099.hnsw").exists());
        assert!(!dir.join("manifest.tmp").exists());
        fs::remove_dir_all(dir).unwrap();
    }
}

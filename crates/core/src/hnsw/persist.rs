//! On-disk format for an HNSW index.
//!
//! ```text
//! 0    header (64 bytes)
//!        magic "VECHNSW1", version u32, dim u32, metric u32, m u32,
//!        ef_construction u32, entry_level u32 (u32::MAX if empty), count u64,
//!        entry_id u32, pad u32, upper_len u64, rng_state u64
//! 64   data           f32 x count*dim
//!      levels         u8  x count
//!      links0         u32 x count*(2m+1)
//!      upper_offsets  u64 x count+1
//!      upper          u32 x upper_len
//! end  crc32 of bytes [0, end) as u64
//! ```
//!
//! Every section starts on a 64-byte boundary, so loading is a single mmap and
//! each array is used in place. Files are written to a temp path, fsynced, then
//! renamed, so a crash leaves either the old file or the new one.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use super::*;
use crate::buf::as_bytes;

const MAGIC: &[u8; 8] = b"VECHNSW1";
const VERSION: u32 = 1;
const HEADER: usize = 64;
const ALIGN: usize = 64;

struct Layout {
    data: usize,
    levels: usize,
    links0: usize,
    upper_offsets: usize,
    upper: usize,
    end: usize,
}

fn align(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

fn layout(count: usize, dim: usize, m0: usize, upper_len: usize) -> Layout {
    let data = HEADER;
    let levels = align(data + count * dim * 4);
    let links0 = align(levels + count);
    let upper_offsets = align(links0 + count * (m0 + 1) * 4);
    let upper = align(upper_offsets + (count + 1) * 8);
    let end = align(upper + upper_len * 4);
    Layout {
        data,
        levels,
        links0,
        upper_offsets,
        upper,
        end,
    }
}

fn metric_code(m: Metric) -> u32 {
    match m {
        Metric::L2 => 0,
        Metric::InnerProduct => 1,
        Metric::Cosine => 2,
    }
}

fn metric_from_code(c: u32) -> Option<Metric> {
    match c {
        0 => Some(Metric::L2),
        1 => Some(Metric::InnerProduct),
        2 => Some(Metric::Cosine),
        _ => None,
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Writes sequentially, padding to section offsets and checksumming as it goes.
struct SectionWriter<W: Write> {
    inner: W,
    crc: crc32fast::Hasher,
    pos: usize,
}

impl<W: Write> SectionWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.crc.update(bytes);
        self.pos += bytes.len();
        self.inner.write_all(bytes)
    }

    fn pad_to(&mut self, offset: usize) -> io::Result<()> {
        debug_assert!(offset >= self.pos);
        self.write(&vec![0u8; offset - self.pos])
    }
}

/// Fsyncs a directory so a rename inside it is durable.
pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

impl Hnsw {
    /// Writes the index to `path` atomically (temp file, fsync, rename).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let count = self.len();
        let upper_len = self.upper.len();
        let l = layout(count, self.dim, self.m0, upper_len);
        let (entry_id, entry_level) = self.entry.map_or((0, u32::MAX), |e| (e.id, e.level as u32));

        let mut header = [0u8; HEADER];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&(self.dim as u32).to_le_bytes());
        header[16..20].copy_from_slice(&metric_code(self.metric).to_le_bytes());
        header[20..24].copy_from_slice(&(self.params.m as u32).to_le_bytes());
        header[24..28].copy_from_slice(&(self.params.ef_construction as u32).to_le_bytes());
        header[28..32].copy_from_slice(&entry_level.to_le_bytes());
        header[32..40].copy_from_slice(&(count as u64).to_le_bytes());
        header[40..44].copy_from_slice(&entry_id.to_le_bytes());
        header[48..56].copy_from_slice(&(upper_len as u64).to_le_bytes());
        header[56..64].copy_from_slice(&self.rng.0.to_le_bytes());

        let tmp = path.with_extension("tmp");
        let file = File::create(&tmp)?;
        let mut w = SectionWriter {
            inner: BufWriter::with_capacity(1 << 20, &file),
            crc: crc32fast::Hasher::new(),
            pos: 0,
        };
        w.write(&header)?;
        w.write(as_bytes(&self.data))?;
        w.pad_to(l.levels)?;
        w.write(as_bytes(&self.levels))?;
        w.pad_to(l.links0)?;
        w.write(as_bytes(&self.links0))?;
        w.pad_to(l.upper_offsets)?;
        w.write(as_bytes(&self.upper_offsets))?;
        w.pad_to(l.upper)?;
        w.write(as_bytes(&self.upper))?;
        w.pad_to(l.end)?;
        let SectionWriter { mut inner, crc, .. } = w;
        inner.write_all(&(crc.finalize() as u64).to_le_bytes())?;
        inner.flush()?;
        drop(inner);
        file.sync_all()?;

        fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            sync_dir(dir)?;
        }
        Ok(())
    }

    /// Opens an index file with mmap. The arrays are used in place; the first
    /// insert after loading copies them into memory.
    ///
    /// `verify` checks the crc32 of the whole file, which reads every page.
    /// Skip it for the fastest cold start when the file is known to be good.
    pub fn load(path: impl AsRef<Path>, verify: bool) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: index files are never modified in place (saves go through a
        // rename), so the mapped bytes can't change underneath us.
        let map = unsafe { Mmap::map(&file)? };
        let name = path.display();

        if map.len() < HEADER || &map[0..8] != MAGIC {
            return Err(invalid(format!("{name}: not an index file")));
        }
        let u32_at = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap());

        let version = u32_at(8);
        if version != VERSION {
            return Err(invalid(format!("{name}: unsupported version {version}")));
        }
        let dim = u32_at(12) as usize;
        let metric = metric_from_code(u32_at(16)).ok_or_else(|| invalid(format!("{name}: bad metric")))?;
        let m = u32_at(20) as usize;
        let ef_construction = u32_at(24) as usize;
        let entry_level = u32_at(28);
        let count = u64_at(32) as usize;
        let entry_id = u32_at(40);
        let upper_len = u64_at(48) as usize;
        let rng_state = u64_at(56);
        if dim == 0 || m < 2 {
            return Err(invalid(format!("{name}: bad header")));
        }

        let m0 = 2 * m;
        let l = layout(count, dim, m0, upper_len);
        if map.len() != l.end + 8 {
            return Err(invalid(format!(
                "{name}: expected {} bytes, found {} (truncated?)",
                l.end + 8,
                map.len()
            )));
        }
        if verify {
            let stored = u64_at(l.end);
            if crc32fast::hash(&map[..l.end]) as u64 != stored {
                return Err(invalid(format!("{name}: checksum mismatch")));
            }
        }

        // Searches touch pages in random order, so without this the first queries
        // after a load stall on one page fault after another. WillNeed starts the
        // kernel reading the whole file in the background; load still returns at once.
        let _ = map.advise(memmap2::Advice::WillNeed);
        let map = Arc::new(map);
        let upper_offsets: Buf<u64> = Buf::mapped(map.clone(), l.upper_offsets, count + 1);
        if upper_offsets[count] as usize != upper_len {
            return Err(invalid(format!("{name}: inconsistent upper-layer offsets")));
        }
        let entry = (entry_level != u32::MAX).then_some(Entry {
            id: entry_id,
            level: entry_level as usize,
        });
        if entry.is_some_and(|e| e.id as usize >= count) {
            return Err(invalid(format!("{name}: entry point out of range")));
        }

        Ok(Self {
            dim,
            metric,
            params: HnswParams {
                m,
                ef_construction,
                seed: 0,
            },
            m0,
            ml: 1.0 / (m as f64).ln(),
            data: Buf::mapped(map.clone(), l.data, count * dim),
            levels: Buf::mapped(map.clone(), l.levels, count),
            links0: Buf::mapped(map.clone(), l.links0, count * (m0 + 1)),
            upper: Buf::mapped(map, l.upper, upper_len),
            upper_offsets,
            locks: (0..count).map(|_| Mutex::new(())).collect(),
            entry,
            rng: SplitMix64(rng_state),
            visited: Mutex::new(Vec::new()),
        })
    }

    /// Whether the index is still reading from its file rather than memory.
    pub fn is_mapped(&self) -> bool {
        self.data.is_mapped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        (0..n * dim).map(|_| rng.next_f64() as f32 * 2.0 - 1.0).collect()
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vecengine-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn roundtrip_gives_identical_results() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("index.hnsw");
        let dim = 12;
        let mut index = Hnsw::new(dim, Metric::Cosine, HnswParams::default());
        index.add_batch(&random_data(2000, dim, 1));
        index.save(&path).unwrap();

        let loaded = Hnsw::load(&path, true).unwrap();
        assert!(loaded.is_mapped());
        assert_eq!(loaded.len(), 2000);
        assert_eq!(loaded.metric(), Metric::Cosine);
        for q in random_data(20, dim, 2).chunks_exact(dim) {
            assert_eq!(index.search(q, 10, 50), loaded.search(q, 10, 50));
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loaded_index_accepts_inserts() {
        let dir = temp_dir("grow");
        let path = dir.join("index.hnsw");
        let dim = 8;
        let data = random_data(1500, dim, 3);
        let mut index = Hnsw::new(dim, Metric::L2, HnswParams::default());
        index.add_batch(&data[..1000 * dim]);
        index.save(&path).unwrap();

        let mut loaded = Hnsw::load(&path, true).unwrap();
        loaded.add_batch(&data[1000 * dim..]);
        assert!(!loaded.is_mapped());
        assert_eq!(loaded.len(), 1500);
        let last = &data[1499 * dim..];
        assert_eq!(loaded.search(last, 1, 50)[0].id, 1499);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_index_roundtrips() {
        let dir = temp_dir("empty");
        let path = dir.join("index.hnsw");
        Hnsw::new(4, Metric::L2, HnswParams::default())
            .save(&path)
            .unwrap();
        let loaded = Hnsw::load(&path, true).unwrap();
        assert!(loaded.is_empty());
        assert!(loaded.search(&[0.0; 4], 3, 10).is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn detects_corruption_and_truncation() {
        let dir = temp_dir("corrupt");
        let path = dir.join("index.hnsw");
        let mut index = Hnsw::new(4, Metric::L2, HnswParams::default());
        index.add_batch(&random_data(300, 4, 4));
        index.save(&path).unwrap();
        let good = fs::read(&path).unwrap();

        let mut flipped = good.clone();
        flipped[200] ^= 0xff;
        fs::write(&path, &flipped).unwrap();
        let err = Hnsw::load(&path, true).err().unwrap();
        assert!(err.to_string().contains("checksum"), "{err}");

        fs::write(&path, &good[..good.len() - 100]).unwrap();
        let err = Hnsw::load(&path, true).err().unwrap();
        assert!(err.to_string().contains("truncated"), "{err}");

        fs::write(&path, b"not an index").unwrap();
        assert!(Hnsw::load(&path, true).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}

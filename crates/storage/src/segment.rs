//! A segment: an HNSW index plus the external ids, payloads and deletion bitmap
//! of its points. The memtable is a segment that hasn't been written yet.
//!
//! Files for segment N:
//!   seg-N.hnsw   graph and vectors (see vecengine::hnsw)
//!   seg-N.meta   external ids and payloads
//!   seg-N.del    deletion bitmap, absent if nothing was deleted
//!
//! .meta and .del end in a crc32 and are replaced atomically.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use vecengine::{Hnsw, HnswParams, Metric};

use crate::bitset::Bitset;
use crate::filter::PayloadIndex;

pub(crate) struct Segment {
    /// None for the memtable.
    pub id: Option<u64>,
    pub index: Hnsw,
    pub ids: Vec<u64>,
    pub payloads: Vec<Vec<u8>>,
    pub deleted: Bitset,
    pub payload_index: PayloadIndex,
    /// Deletions since the .del file was last written.
    pub deleted_dirty: bool,
}

impl Segment {
    pub fn empty(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self {
            id: None,
            index: Hnsw::new(dim, metric, params),
            ids: Vec::new(),
            payloads: Vec::new(),
            deleted: Bitset::default(),
            payload_index: PayloadIndex::default(),
            deleted_dirty: false,
        }
    }

    pub fn is_live(&self, i: u32) -> bool {
        !self.deleted.get(i as usize)
    }

    pub fn live_count(&self) -> usize {
        self.ids.len() - self.deleted.count()
    }

    pub fn load(dir: &Path, id: u64) -> io::Result<Self> {
        let index = Hnsw::load(path(dir, id, "hnsw"), true)?;
        let (ids, payloads) = decode_meta(&read_checked(&path(dir, id, "meta"))?)?;
        let deleted = match read_checked(&path(dir, id, "del")) {
            Ok(bytes) => Bitset::decode(&bytes)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Bitset::default(),
            Err(e) => return Err(e),
        };
        if ids.len() != index.len() {
            return Err(invalid(format!(
                "segment {id}: index has {} points, meta has {}",
                index.len(),
                ids.len()
            )));
        }
        let payload_index = PayloadIndex::build(payloads.iter().map(Vec::as_slice));
        Ok(Self {
            id: Some(id),
            index,
            ids,
            payloads,
            deleted,
            payload_index,
            deleted_dirty: false,
        })
    }

    /// Writes the index and meta files for a memtable being turned into segment `id`.
    pub fn write_new(&mut self, dir: &Path, id: u64) -> io::Result<()> {
        self.index.save(path(dir, id, "hnsw"))?;
        write_atomic(
            &path(dir, id, "meta"),
            &with_crc(encode_meta(&self.ids, &self.payloads)),
        )?;
        self.id = Some(id);
        self.deleted_dirty = self.deleted.count() > 0;
        self.payload_index.finish();
        Ok(())
    }

    /// Persists the deletion bitmap if it changed.
    pub fn write_deletes(&mut self, dir: &Path) -> io::Result<()> {
        if let (Some(id), true) = (self.id, self.deleted_dirty) {
            write_atomic(&path(dir, id, "del"), &with_crc(self.deleted.encode()))?;
            self.deleted_dirty = false;
        }
        Ok(())
    }

    pub fn remove_files(dir: &Path, id: u64) {
        for ext in ["hnsw", "meta", "del"] {
            let _ = fs::remove_file(path(dir, id, ext));
        }
    }
}

pub(crate) fn path(dir: &Path, id: u64, ext: &str) -> PathBuf {
    dir.join(format!("seg-{id:06}.{ext}"))
}

/// The segment number in a file name like `seg-000012.hnsw`.
pub(crate) fn parse_segment_file(name: &str) -> Option<u64> {
    name.strip_prefix("seg-")?.split('.').next()?.parse().ok()
}

fn encode_meta(ids: &[u64], payloads: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + ids.len() * 12);
    out.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for p in payloads {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

fn decode_meta(bytes: &[u8]) -> io::Result<(Vec<u64>, Vec<Vec<u8>>)> {
    let bad = || invalid("malformed segment meta");
    let mut pos = 0usize;
    let mut take = |n: usize| -> io::Result<&[u8]> {
        let end = pos.checked_add(n).filter(|&e| e <= bytes.len()).ok_or_else(bad)?;
        let s = &bytes[pos..end];
        pos = end;
        Ok(s)
    };
    let count = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
    let ids = take(count.checked_mul(8).ok_or_else(bad)?)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    let mut payloads = Vec::with_capacity(count.min(bytes.len()));
    for _ in 0..count {
        let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        payloads.push(take(len)?.to_vec());
    }
    if pos != bytes.len() {
        return Err(bad());
    }
    Ok((ids, payloads))
}

fn with_crc(mut bytes: Vec<u8>) -> Vec<u8> {
    let crc = crc32fast::hash(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    bytes
}

fn read_checked(path: &Path) -> io::Result<Vec<u8>> {
    let mut bytes = fs::read(path)?;
    if bytes.len() < 4 {
        return Err(invalid(format!("{}: too short", path.display())));
    }
    let crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    bytes.truncate(bytes.len() - 4);
    if crc32fast::hash(&bytes) != crc {
        return Err(invalid(format!("{}: checksum mismatch", path.display())));
    }
    Ok(bytes)
}

/// Replaces `path` with `bytes`: temp file, fsync, rename, fsync directory.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_roundtrip_and_rejects_garbage() {
        let ids = vec![1, u64::MAX, 42];
        let payloads = vec![b"a".to_vec(), vec![], b"{\"x\":1}".to_vec()];
        let bytes = encode_meta(&ids, &payloads);
        assert_eq!(decode_meta(&bytes).unwrap(), (ids, payloads));
        assert!(decode_meta(&bytes[..bytes.len() - 1]).is_err());
        assert!(decode_meta(&[bytes.clone(), vec![0]].concat()).is_err());
    }

    #[test]
    fn segment_file_names() {
        assert_eq!(parse_segment_file("seg-000012.hnsw"), Some(12));
        assert_eq!(parse_segment_file("seg-000012.del.tmp"), Some(12));
        assert_eq!(parse_segment_file("manifest.json"), None);
    }
}

//! Write-ahead log.
//!
//! Each record is `[len u32][crc32 u32][body]`, body is
//! `op u8, id u64` then for upserts `dim u32, f32 x dim, payload_len u32, payload`.
//! A crash can leave a partial record at the tail; replay stops at the first
//! record that is short or fails its checksum and truncates the file there.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Upsert {
        id: u64,
        vector: Vec<f32>,
        payload: Vec<u8>,
    },
    Delete {
        id: u64,
    },
}

const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;
/// Upper bound on a single record, to reject garbage lengths before allocating.
const MAX_RECORD: usize = 64 << 20;

impl Record {
    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&[0u8; 8]); // len + crc, filled in below
        match self {
            Record::Upsert { id, vector, payload } => {
                out.push(OP_UPSERT);
                out.extend_from_slice(&id.to_le_bytes());
                out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for x in vector {
                    out.extend_from_slice(&x.to_le_bytes());
                }
                out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                out.extend_from_slice(payload);
            }
            Record::Delete { id } => {
                out.push(OP_DELETE);
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
        let body = &out[start + 8..];
        let len = body.len() as u32;
        let crc = crc32fast::hash(body);
        out[start..start + 4].copy_from_slice(&len.to_le_bytes());
        out[start + 4..start + 8].copy_from_slice(&crc.to_le_bytes());
    }

    fn decode(body: &[u8]) -> Option<Self> {
        let mut r = body;
        let op = take(&mut r, 1)?[0];
        let id = u64::from_le_bytes(take(&mut r, 8)?.try_into().ok()?);
        let rec = match op {
            OP_UPSERT => {
                let dim = u32::from_le_bytes(take(&mut r, 4)?.try_into().ok()?) as usize;
                let vector = take(&mut r, dim.checked_mul(4)?)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                let plen = u32::from_le_bytes(take(&mut r, 4)?.try_into().ok()?) as usize;
                let payload = take(&mut r, plen)?.to_vec();
                Record::Upsert { id, vector, payload }
            }
            OP_DELETE => Record::Delete { id },
            _ => return None,
        };
        r.is_empty().then_some(rec)
    }
}

fn take<'a>(r: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if r.len() < n {
        return None;
    }
    let (head, tail) = r.split_at(n);
    *r = tail;
    Some(head)
}

pub struct Wal {
    file: File,
    sync: bool,
    buf: Vec<u8>,
}

impl Wal {
    /// Opens (creating if needed) the log at `path`, returns it along with every
    /// intact record. A torn tail is cut off.
    pub fn open(path: &Path, sync: bool) -> io::Result<(Self, Vec<Record>)> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let mut records = Vec::new();
        let mut valid = 0u64;
        {
            let mut reader = BufReader::new(&file);
            let mut head = [0u8; 8];
            loop {
                match reader.read_exact(&mut head) {
                    Ok(()) => {}
                    Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
                let len = u32::from_le_bytes(head[0..4].try_into().unwrap()) as usize;
                let crc = u32::from_le_bytes(head[4..8].try_into().unwrap());
                if len > MAX_RECORD {
                    break;
                }
                let mut body = vec![0u8; len];
                if reader.read_exact(&mut body).is_err() || crc32fast::hash(&body) != crc {
                    break;
                }
                let Some(rec) = Record::decode(&body) else { break };
                records.push(rec);
                valid += 8 + len as u64;
            }
        }

        if file.metadata()?.len() != valid {
            file.set_len(valid)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::Start(valid))?;
        Ok((
            Self {
                file,
                sync,
                buf: Vec::new(),
            },
            records,
        ))
    }

    /// Appends records as one write. Once this returns (with `sync` on) they
    /// survive a crash.
    pub fn append(&mut self, records: &[Record]) -> io::Result<()> {
        self.buf.clear();
        for r in records {
            r.encode(&mut self.buf);
        }
        self.file.write_all(&self.buf)?;
        if self.sync {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Empties the log, after its contents have been made durable elsewhere.
    pub fn reset(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vecengine-wal-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("wal.log")
    }

    fn sample() -> Vec<Record> {
        vec![
            Record::Upsert {
                id: 7,
                vector: vec![1.0, -2.5, 3.25],
                payload: br#"{"tag":"a"}"#.to_vec(),
            },
            Record::Delete { id: 3 },
            Record::Upsert {
                id: u64::MAX,
                vector: vec![],
                payload: vec![],
            },
        ]
    }

    #[test]
    fn replays_what_was_appended() {
        let path = temp_path("replay");
        let (mut wal, recs) = Wal::open(&path, true).unwrap();
        assert!(recs.is_empty());
        wal.append(&sample()[..2]).unwrap();
        wal.append(&sample()[2..]).unwrap();
        drop(wal);

        let (_, recs) = Wal::open(&path, true).unwrap();
        assert_eq!(recs, sample());
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn torn_tail_is_truncated_at_every_cut_point() {
        let path = temp_path("torn");
        let (mut wal, _) = Wal::open(&path, true).unwrap();
        wal.append(&sample()).unwrap();
        drop(wal);
        let full = std::fs::read(&path).unwrap();

        let mut good_prefixes = vec![0usize];
        let mut buf = Vec::new();
        for r in sample() {
            r.encode(&mut buf);
            good_prefixes.push(buf.len());
        }

        for cut in 0..full.len() {
            std::fs::write(&path, &full[..cut]).unwrap();
            let (mut wal, recs) = Wal::open(&path, true).unwrap();
            let expected = good_prefixes.iter().filter(|&&p| p <= cut).count() - 1;
            assert_eq!(recs, sample()[..expected], "cut at {cut}");
            assert_eq!(
                std::fs::metadata(&path).unwrap().len() as usize,
                good_prefixes[expected]
            );

            // The log stays appendable after recovery.
            wal.append(&[Record::Delete { id: 1 }]).unwrap();
            drop(wal);
            let (_, recs) = Wal::open(&path, true).unwrap();
            assert_eq!(recs.last(), Some(&Record::Delete { id: 1 }));
        }
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn corrupt_record_stops_replay() {
        let path = temp_path("corrupt");
        let (mut wal, _) = Wal::open(&path, true).unwrap();
        wal.append(&sample()).unwrap();
        drop(wal);
        let mut bytes = std::fs::read(&path).unwrap();
        let second_record = {
            let mut b = Vec::new();
            sample()[0].encode(&mut b);
            b.len()
        };
        bytes[second_record + 9] ^= 1;
        std::fs::write(&path, &bytes).unwrap();

        let (_, recs) = Wal::open(&path, true).unwrap();
        assert_eq!(recs, sample()[..1]);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn reset_empties_the_log() {
        let path = temp_path("reset");
        let (mut wal, _) = Wal::open(&path, true).unwrap();
        wal.append(&sample()).unwrap();
        wal.reset().unwrap();
        wal.append(&[Record::Delete { id: 9 }]).unwrap();
        drop(wal);
        let (_, recs) = Wal::open(&path, true).unwrap();
        assert_eq!(recs, vec![Record::Delete { id: 9 }]);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}

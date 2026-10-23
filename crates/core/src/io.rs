//! Readers for the fvecs / ivecs format written by `bench/download.py`.
//!
//! Each row is a little-endian `i32` dimension followed by `dim` 4-byte values.

use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::path::Path;

/// Row-major matrix of `len() x dim` values.
#[derive(Debug, Clone)]
pub struct Vectors<T> {
    pub dim: usize,
    pub data: Vec<T>,
}

impl<T> Vectors<T> {
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> std::slice::ChunksExact<'_, T> {
        self.data.chunks_exact(self.dim.max(1))
    }
}

pub fn read_fvecs(path: impl AsRef<Path>) -> io::Result<Vectors<f32>> {
    read_vecs(path.as_ref(), f32::from_le_bytes)
}

pub fn read_ivecs(path: impl AsRef<Path>) -> io::Result<Vectors<u32>> {
    read_vecs(path.as_ref(), u32::from_le_bytes)
}

fn read_vecs<T>(path: &Path, decode: fn([u8; 4]) -> T) -> io::Result<Vectors<T>> {
    let file = File::open(path)?;
    let size = file.metadata()?.len() as usize;
    let mut reader = BufReader::with_capacity(1 << 20, file);

    let mut header = [0u8; 4];
    if let Err(e) = reader.read_exact(&mut header) {
        return if e.kind() == ErrorKind::UnexpectedEof {
            Ok(Vectors {
                dim: 0,
                data: Vec::new(),
            })
        } else {
            Err(e)
        };
    }
    let dim = i32::from_le_bytes(header);
    if dim <= 0 {
        return Err(invalid(format!("{}: bad dimension {dim}", path.display())));
    }
    let dim = dim as usize;
    let row_bytes = 4 * (dim + 1);
    if !size.is_multiple_of(row_bytes) {
        return Err(invalid(format!(
            "{}: size {size} is not a multiple of row size {row_bytes}",
            path.display()
        )));
    }

    let rows = size / row_bytes;
    let mut data = Vec::with_capacity(rows * dim);
    let mut buf = vec![0u8; dim * 4];
    for row in 0..rows {
        if row > 0 {
            reader.read_exact(&mut header)?;
            if i32::from_le_bytes(header) as usize != dim {
                return Err(invalid(format!(
                    "{}: row {row} has a different dimension",
                    path.display()
                )));
            }
        }
        reader.read_exact(&mut buf)?;
        data.extend(buf.as_chunks::<4>().0.iter().map(|b| decode(*b)));
    }
    Ok(Vectors { dim, data })
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvecs(path: &Path, rows: &[Vec<f32>]) {
        let mut f = File::create(path).unwrap();
        for r in rows {
            f.write_all(&(r.len() as i32).to_le_bytes()).unwrap();
            for x in r {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("vecengine-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.fvecs");
        let rows = vec![vec![1.0, 2.0, 3.0], vec![-4.0, 5.5, 0.0]];
        write_fvecs(&path, &rows);

        let v = read_fvecs(&path).unwrap();
        assert_eq!(v.dim, 3);
        assert_eq!(v.len(), 2);
        assert_eq!(v.row(1), &[-4.0, 5.5, 0.0]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = std::env::temp_dir().join(format!("vecengine-io-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.fvecs");
        write_fvecs(&path, &[vec![1.0, 2.0]]);
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0u8; 3]).unwrap();

        assert!(read_fvecs(&path).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

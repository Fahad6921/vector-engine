//! A growable bitmap with a cached population count.

use std::io;

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bitset {
    words: Vec<u64>,
    count: usize,
}

impl Bitset {
    /// All of `0..n` set.
    pub fn full(n: usize) -> Self {
        let mut words = vec![u64::MAX; n / 64];
        if !n.is_multiple_of(64) {
            words.push((1u64 << (n % 64)) - 1);
        }
        Self { words, count: n }
    }

    pub fn get(&self, i: usize) -> bool {
        self.words.get(i / 64).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }

    pub fn set(&mut self, i: usize) {
        if self.words.len() <= i / 64 {
            self.words.resize(i / 64 + 1, 0);
        }
        let bit = 1 << (i % 64);
        if self.words[i / 64] & bit == 0 {
            self.words[i / 64] |= bit;
            self.count += 1;
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn and(&self, other: &Self) -> Self {
        Self::from_words(self.words.iter().zip(&other.words).map(|(a, b)| a & b).collect())
    }

    pub fn or(&self, other: &Self) -> Self {
        let (long, short) = if self.words.len() >= other.words.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut words = long.words.clone();
        for (w, s) in words.iter_mut().zip(&short.words) {
            *w |= s;
        }
        Self::from_words(words)
    }

    /// Bits in `self` that are not in `other`.
    pub fn and_not(&self, other: &Self) -> Self {
        Self::from_words(
            self.words
                .iter()
                .enumerate()
                .map(|(i, a)| a & !other.words.get(i).copied().unwrap_or(0))
                .collect(),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let bit = w.trailing_zeros();
                w &= w - 1;
                Some((wi * 64) as u32 + bit)
            })
        })
    }

    fn from_words(words: Vec<u64>) -> Self {
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Self { words, count }
    }

    pub fn encode(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if !bytes.len().is_multiple_of(8) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bitmap has a partial word",
            ));
        }
        Ok(Self::from_words(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from(ids: &[usize]) -> Bitset {
        let mut b = Bitset::default();
        ids.iter().for_each(|&i| b.set(i));
        b
    }

    #[test]
    fn set_get_count_roundtrip() {
        let b = from(&[0, 5, 63, 64, 200, 5]);
        assert_eq!(b.count(), 5);
        assert!(b.get(64) && !b.get(65) && !b.get(10_000));
        assert_eq!(Bitset::decode(&b.encode()).unwrap(), b);
    }

    #[test]
    fn set_operations() {
        let a = from(&[1, 2, 3, 100]);
        let b = from(&[2, 3, 4]);
        assert_eq!(a.and(&b).iter().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(a.or(&b).iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 100]);
        assert_eq!(b.or(&a).count(), 5);
        assert_eq!(a.and_not(&b).iter().collect::<Vec<_>>(), vec![1, 100]);
        assert_eq!(Bitset::full(70).and_not(&a).count(), 67);
        assert_eq!(Bitset::full(64).count(), 64);
        assert_eq!(Bitset::full(130).iter().last(), Some(129));
    }
}

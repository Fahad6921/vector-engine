//! BM25 keyword search and reciprocal rank fusion.
//!
//! Corpus statistics (document count, average length, document frequencies) are
//! passed in at query time rather than read from the index, so a collection made
//! of several segments can score every segment against the same global numbers.

use std::collections::HashMap;
use std::hash::Hash;

use crate::neighbor::{Neighbor, TopK};

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "which", "with",
];

/// Lowercased alphanumeric runs, minus a small English stopword list.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

#[derive(Default)]
pub struct Bm25Index {
    /// term -> (doc, term frequency), in insertion order.
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    docs: usize,
    total_len: u64,
}

/// Corpus-wide numbers BM25 needs; sum them over segments with `merge`.
#[derive(Debug, Default, Clone)]
pub struct CorpusStats {
    pub docs: usize,
    pub total_len: u64,
    pub df: HashMap<String, usize>,
}

impl CorpusStats {
    pub fn merge(&mut self, other: &CorpusStats) {
        self.docs += other.docs;
        self.total_len += other.total_len;
        for (t, n) in &other.df {
            *self.df.entry(t.clone()).or_default() += n;
        }
    }

    fn avg_len(&self) -> f32 {
        if self.docs == 0 {
            1.0
        } else {
            self.total_len as f32 / self.docs as f32
        }
    }

    fn idf(&self, term: &str) -> f32 {
        let df = self.df.get(term).copied().unwrap_or(0) as f32;
        (1.0 + (self.docs as f32 - df + 0.5) / (df + 0.5)).ln()
    }
}

impl Bm25Index {
    /// Indexes `text` as document `doc`. Each doc should be added once.
    pub fn add(&mut self, doc: u32, text: &str) {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t).or_default() += 1;
        }
        for (t, n) in tf {
            self.postings.entry(t.to_string()).or_default().push((doc, n));
        }
        if self.doc_len.len() <= doc as usize {
            self.doc_len.resize(doc as usize + 1, 0);
        }
        self.doc_len[doc as usize] = tokens.len() as u32;
        self.docs += 1;
        self.total_len += tokens.len() as u64;
    }

    /// This index's contribution to corpus statistics for the given query terms.
    pub fn stats(&self, terms: &[String]) -> CorpusStats {
        CorpusStats {
            docs: self.docs,
            total_len: self.total_len,
            df: terms
                .iter()
                .map(|t| (t.clone(), self.postings.get(t).map_or(0, Vec::len)))
                .collect(),
        }
    }

    /// Top `k` docs by BM25 for already-tokenized `terms`, highest score first.
    /// `Neighbor::distance` holds the negated score so lower is better, as elsewhere.
    pub fn search<F: Fn(u32) -> bool>(
        &self,
        terms: &[String],
        k: usize,
        stats: &CorpusStats,
        allowed: F,
    ) -> Vec<Neighbor> {
        let avg = stats.avg_len();
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for t in terms {
            let Some(list) = self.postings.get(t) else {
                continue;
            };
            let idf = stats.idf(t);
            for &(doc, tf) in list {
                let tf = tf as f32;
                let norm = 1.0 - B + B * self.doc_len[doc as usize] as f32 / avg;
                *scores.entry(doc).or_default() += idf * tf * (K1 + 1.0) / (tf + K1 * norm);
            }
        }
        let mut top = TopK::new(k);
        for (doc, s) in scores {
            if allowed(doc) {
                top.push(Neighbor::new(doc, -s));
            }
        }
        top.into_sorted_vec()
    }
}

/// Reciprocal rank fusion: each item scores sum(1 / (c + rank)) over the lists it
/// appears in (rank from 1). Only ranks are used, so lists with incomparable
/// scores (BM25 vs vector distance) combine without tuning. Highest first.
pub fn rrf<T: Copy + Eq + Hash + Ord>(lists: &[Vec<T>], c: f32) -> Vec<(T, f32)> {
    let mut scores: HashMap<T, f32> = HashMap::new();
    for list in lists {
        for (rank, &item) in list.iter().enumerate() {
            *scores.entry(item).or_default() += 1.0 / (c + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(T, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer() {
        assert_eq!(
            tokenize("The COVID-19 vaccine's efficacy, in 2021!"),
            vec!["covid", "19", "vaccine", "s", "efficacy", "2021"]
        );
        assert!(tokenize("  the and of ").is_empty());
    }

    fn index() -> Bm25Index {
        let mut idx = Bm25Index::default();
        idx.add(0, "rust vector database written in rust");
        idx.add(1, "a python vector library");
        idx.add(2, "cooking pasta at home");
        idx.add(4, "rust compiler internals"); // gap at 3: no text
        idx
    }

    fn ids(res: &[Neighbor]) -> Vec<u32> {
        res.iter().map(|n| n.id).collect()
    }

    #[test]
    fn ranks_by_bm25() {
        let idx = index();
        let q = tokenize("rust vector");
        let stats = idx.stats(&q);
        let res = idx.search(&q, 10, &stats, |_| true);
        // Doc 0 has both terms, rust twice. Pasta (doc 2) matches nothing.
        assert_eq!(ids(&res)[0], 0);
        assert_eq!(res.len(), 3);
        assert!(res.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn respects_filter_and_k() {
        let idx = index();
        let q = tokenize("rust");
        let stats = idx.stats(&q);
        assert_eq!(ids(&idx.search(&q, 10, &stats, |d| d != 0)), vec![4]);
        assert_eq!(idx.search(&q, 1, &stats, |_| true).len(), 1);
        assert!(idx.search(&tokenize("nothing"), 5, &stats, |_| true).is_empty());
    }

    #[test]
    fn merged_stats_match_single_index() {
        let docs = ["red apple", "green apple", "red car", "blue car", "apple pie"];
        let mut whole = Bm25Index::default();
        let (mut a, mut b) = (Bm25Index::default(), Bm25Index::default());
        for (i, d) in docs.iter().enumerate() {
            whole.add(i as u32, d);
            if i < 3 {
                a.add(i as u32, d)
            } else {
                b.add(i as u32, d)
            }
        }
        let q = tokenize("red apple");
        let mut merged = a.stats(&q);
        merged.merge(&b.stats(&q));
        let mut split: Vec<Neighbor> = a.search(&q, 10, &merged, |_| true);
        split.extend(b.search(&q, 10, &merged, |_| true));
        split.sort_unstable();
        assert_eq!(split, whole.search(&q, 10, &whole.stats(&q), |_| true));
    }

    #[test]
    fn rrf_combines_ranks() {
        let fused = rrf(&[vec![1, 2, 3], vec![3, 1, 4]], 60.0);
        let order: Vec<i32> = fused.iter().map(|x| x.0).collect();
        // 1: 1/61 + 1/62; 3: 1/63 + 1/61; 2: 1/62; 4: 1/63
        assert_eq!(order, vec![1, 3, 2, 4]);
    }
}
